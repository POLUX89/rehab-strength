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

# La app de notificación es un applet de Automator; macOS ya le concedió permisos.
# Se puede reubicar con REHAB_NOTIFY_APP en el .env.
NOTIFY_APP="${REHAB_NOTIFY_APP:-$HOME/Desktop/GYM WORKOUT ANALYSIS PROJECT/NotifyGymPipeline.app}"

notify() {
  if [ -d "$NOTIFY_APP" ]; then
    open "$NOTIFY_APP" 2>/dev/null && return
  fi
  # Sin el applet, macOS puede pedir permiso para estas notificaciones.
  osascript -e "display notification \"$1\" with title \"🏋️ Rehab Strength\"" 2>/dev/null
}

if [ ! -x ".venv/bin/python" ]; then
  echo "❌ No existe .venv/bin/python. Corré: make setup" >>"$LOG"
  osascript -e 'display notification "Falta el venv — corré make setup" with title "🏋️ Rehab Strength"' 2>/dev/null
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
  osascript -e 'display notification "La ingesta falló — revisá el log" with title "🏋️ Rehab Strength"' 2>/dev/null
fi

exit $rc
