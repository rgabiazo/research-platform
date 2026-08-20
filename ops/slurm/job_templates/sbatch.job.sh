
#!/usr/bin/env bash
#SBATCH --job-name={{ job_name }}
#SBATCH --cpus-per-task={{ cpus }}
{{ mem_directive }}
#SBATCH --time={{ time }}
#SBATCH --output={{ log_out }}
#SBATCH --error={{ log_err }}
{{ optional_directives }}

set -euo pipefail

{{ command }}
