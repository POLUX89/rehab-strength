#!/bin/zsh
# Corre la ingesta completa y avisa por notificación de macOS.
#
# Pensado para dispararse solo al despertar el Mac, vía sleepwatcher:
#   ~/.wakeup  ->  este script
#
# A mano:  ./scripts/run_pipeline.sh
# Log:     ~/Library/Logs/rehab_strength_pipeline.log

set -uo pipefail

# Raíz del repo derivada de la ubicación de este script: nada de rutas absolutas.
REPO="${0:A:h:h}"
cd "$REPO" || exit 1

LOG="$HOME/Library/Logs/rehab_strength_pipeline.log"
mkdir -p "$(dirname "$LOG")"
echo "---- $(date) ---- ingesta automática (wake)" >>"$LOG"

# Notificación nativa de macOS, sin depender de ninguna app externa.
# La primera vez macOS puede pedir permiso para notificaciones del intérprete.
notify() {
  osascript -e "display notification \"$1\" with title \"🏋️ Rehab Strength\"" 2>/dev/null
}

if [ ! -x ".venv/bin/python" ]; then
  echo "❌ No existe .venv/bin/python. Corré: make setup" >>"$LOG"
  notify "Falta el venv — corré make setup"
  exit 1
fi

# 'status' es de solo lectura en zsh: usar otro nombre.
.venv/bin/python -m rehab_strength.ingest.run_all >>"$LOG" 2>&1
rc=$?

if [ $rc -eq 0 ]; then
  echo "✅ OK" >>"$LOG"
  notify "GYM Pipeline completed successfully!"
else
  # El pipeline viejo solo avisaba al terminar bien: si Google fallaba, silencio.
  echo "❌ Falló con código $rc" >>"$LOG"
  notify "La ingesta falló — revisá el log"
fi

exit $rc
