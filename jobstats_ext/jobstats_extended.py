#!/usr/bin/python3
# jobstats_extended - a LOCAL, non-invasive superset of the stock `jobstats`
# that folds extra DCGM profiling metrics (SM-active, SM-occupancy, tensor-pipe,
# DRAM-active, power) INTO the jobstats blob, keyed per-GPU by minor_number
# exactly like the built-in gpu_utilization. Subclasses jobstats.Jobstats, so it
# reuses all of jobstats' sacct/Prometheus/serialization logic -- the only new
# code is the DCGM augmentation. Lets you test the extended blob end-to-end
# before deciding whether to swap it into the root-level install cluster-wide.
#
# Why a join is needed: the DCGM_FI_* series live in a different exporter than
# nvidia_gpu_duty_cycle and carry `UUID`/`gpu` (not `minor_number`), and the
# `gpu` index != slurm `minor_number`. So we (1) map UUID -> (node, minor) from
# the nvidia_gpu_jobId companion -- the same job->GPU mapping jobstats uses --
# then (2) query each DCGM metric by UUID and store it back under minor_number,
# so the new keys align with gpu_utilization in the blob.
#
# Usage:
#   ./jobstats_extended.py 18583067            # formatted report (like `jobstats`) + DCGM
#   ./jobstats_extended.py --json 18583067     # extended blob as pretty JSON
#   ./jobstats_extended.py -b 18583067         # JS1:<base64> blob (storage form)
#   ./jobstats_extended.py --no-dcgm 18583067  # report/blob without the DCGM metrics
#
# Author: Bala Desinghu, Senior AI/HPC Research Computing Engineer, Kempner Institute, Harvard

import sys, json, argparse

for _p in ("/usr/local/bin", "/usr/bin"):
  if _p not in sys.path:
    sys.path.append(_p)
from config import PROM_SERVER
try:
  from config import SAMPLING_PERIOD
except Exception:
  SAMPLING_PERIOD = 60
from jobstats import Jobstats
from output_formatters import ClassicOutput
import requests

# Extra metrics folded into the blob: (blob_key, prometheus_metric, scale,
# label, unit). scale 100 turns DCGM 0-1 fractions into percent so jobstats'
# round(v,1) keeps precision and they read like gpu_utilization (0-100). All are
# time-averaged over the job window, like the duty cycle. label/unit are only
# used by the human-readable report.
DCGM_EXTRA = [
    ("gpu_sm_active",     "DCGM_FI_PROF_SM_ACTIVE",          100, "SM active",     "%"),
    ("gpu_sm_occupancy",  "DCGM_FI_PROF_SM_OCCUPANCY",       100, "SM occupancy",  "%"),
    ("gpu_tensor_active", "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", 100, "Tensor active", "%"),
    ("gpu_dram_active",   "DCGM_FI_PROF_DRAM_ACTIVE",        100, "DRAM active",   "%"),
    ("gpu_power_usage",   "DCGM_FI_DEV_POWER_USAGE",         1,   "Power usage",   "W"),
]


class JobstatsExtended(Jobstats):
  """jobstats.Jobstats + DCGM profiling metrics in the blob."""

  def __init__(self, *a, dcgm=True, **kw):
    self.dcgm_added = []
    super().__init__(*a, **kw)
    # sp_node is populated by the parent (from the cached blob or a fresh
    # Prometheus pass); augment it after, only for long-enough GPU jobs.
    if dcgm and getattr(self, 'gpus', 0) and self.sp_node:
      try:
        self._augment_with_dcgm()
      except Exception as e:
        self.debug_print("DCGM augmentation skipped: %s" % e)

  def _prom(self, query):
    """Instant query at the job end time -> result list (empty on error)."""
    r = requests.get(self.prom_server + '/api/v1/query',
                     params={'query': query, 'time': self.end}, timeout=60)
    j = r.json()
    return j['data']['result'] if j.get('status') == 'success' else []

  def _augment_with_dcgm(self):
    cl = "slurm_cluster='%s'" % self.cluster if self.cluster else ''
    # 1) UUID -> (node, minor) for this job's GPUs (uuid is not stored in the
    #    blob, so this companion query is always needed).
    disc = self._prom("max_over_time((nvidia_gpu_jobId{%s} == %s)[%ds:])"
                      % (cl, self.jobidraw, self.diff))
    uuid_to = {}
    for s in disc:
      m = s['metric']
      u = m.get('uuid')
      if u and m.get('minor_number') is not None and 'host' in m:
        uuid_to[u] = (m['host'].split(':')[0], m['minor_number'])
    if not uuid_to:
      return
    rx = "^(" + "|".join(uuid_to) + ")$"   # UUIDs are hex+hyphen, RE2-safe
    # 2) each DCGM metric, joined on UUID, stored back under minor_number.
    for key, metric, scale, _label, _unit in DCGM_EXTRA:
      res = self._prom("avg_over_time((%s{UUID=~\"%s\"})[%ds:])" % (metric, rx, self.diff))
      got = False
      for s in res:
        node_minor = uuid_to.get(s['metric'].get('UUID'))
        if not node_minor:
          continue
        try:
          v = round(float(s['value'][1]) * scale, 1)
        except (TypeError, ValueError):
          continue
        node, minor = node_minor
        self.sp_node.setdefault(node, {}).setdefault(key, {})[minor] = v
        got = True
      if got:
        self.dcgm_added.append(key)


def dcgm_section(js, fmt):
  """Render the DCGM profiling metrics folded into js.sp_node as a report
     section in the ClassicOutput style. fmt is the ClassicOutput instance
     already used for the main report (its txt_* colors and draw_meter are
     reused). Returns "" when no DCGM keys are present."""
  nodes = sorted(js.sp_node)
  if not any(key in js.sp_node[nd] for nd in nodes for key, *_ in DCGM_EXTRA):
    return ""
  width, gutter = fmt.width, "  "
  bold, normal = fmt.txt_bold, fmt.txt_normal
  out = "\n"
  out += f"{bold}GPU Profiling Metrics (DCGM){normal}".center(width) + "\n"
  out += width * "=" + "\n"
  out += f"{gutter}Averaged over the run time. SM/tensor/DRAM are % busy; power is watts.\n\n"
  # overall: average each metric across every GPU on every node
  out += f"{gutter}Overall (averaged across GPUs)\n"
  for key, _metric, _scale, label, unit in DCGM_EXTRA:
    vals = [v for nd in nodes if key in js.sp_node[nd]
            for v in js.sp_node[nd][key].values()]
    if not vals:
      continue
    avg = sum(vals) / len(vals)
    if unit == "%":
      out += f"{gutter}  {label:<14}{fmt.draw_meter(int(round(avg)), 'dcgm')}\n"
    else:
      out += f"{gutter}  {label:<14}{avg:.1f}{unit} (avg per GPU)\n"
  # per node / GPU detail
  out += f"\n{gutter}Per node / GPU\n"
  for nd in nodes:
    d = js.sp_node[nd]
    minors = set()
    for key, *_ in DCGM_EXTRA:
      if key in d:
        minors.update(d[key])
    for minor in sorted(minors, key=lambda m: int(m)):
      parts = [f"{label}={d[key][minor]}{unit}"
               for key, _metric, _scale, label, unit in DCGM_EXTRA
               if key in d and minor in d[key]]
      out += f"{gutter}    {nd} (GPU {minor}): " + "  ".join(parts) + "\n"
  return out


if __name__ == "__main__":
  ap = argparse.ArgumentParser(description="jobstats with extra DCGM profiling metrics (formatted report by default)")
  ap.add_argument("jobids", nargs="+", help="one or more Slurm job IDs")
  ap.add_argument("-j", "--json", action="store_true",
                  help="emit the extended blob as pretty JSON instead of the formatted report")
  ap.add_argument("-b", "--base64", action="store_true",
                  help="emit the JS1:<base64 gzip JSON> blob (the AdminComment storage form)")
  ap.add_argument("--no-dcgm", dest="dcgm", action="store_false",
                  help="skip the DCGM augmentation (report/blob without the extra metrics)")
  ap.add_argument("-f", "--force", action="store_true",
                  help="force recompute from Prometheus instead of the cached blob")
  ap.add_argument("-c", "--cluster", default=None, help="cluster (default: from sacct)")
  ap.add_argument("-n", "--no-color", action="store_true",
                  help="render the report without colorization")
  ap.add_argument("-d", "--debug", action="store_true", help="debug output")
  args = ap.parse_args()

  # JSON/base64 output is the raw blob, so let the parent behave like `jobstats
  # -j/-b`; the formatted report needs parse_stats(), so json_or_base64=False.
  raw = args.json or args.base64

  for jid in args.jobids:
    try:
      js = JobstatsExtended(jid, cluster=args.cluster, prom_server=PROM_SERVER,
                            debug=args.debug, force_recalc=args.force,
                            json_or_base64=raw, dcgm=args.dcgm)
    except SystemExit:          # parent .error() exits; keep going for other ids
      continue
    if args.dcgm:
      print("# job %s: added %s" % (jid, ", ".join(js.dcgm_added) or "(none)"),
            file=sys.stderr)
    if args.base64:
      print("JS1:" + js.report_job_json(True))
    elif args.json:
      print(js.report_job_json(False))
    else:
      fmt = ClassicOutput(js)
      report = fmt.output(no_color=args.no_color)
      if args.dcgm:
        report += dcgm_section(js, fmt)
      print(report)
