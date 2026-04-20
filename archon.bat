@echo off
:: Archon CLI launcher for Windows
:: Place this file somewhere on your PATH (e.g. C:\Windows\System32 or a custom bin dir)

set "SCRIPT_DIR=%~dp0"
python "%SCRIPT_DIR%archon\cli.py" %*
