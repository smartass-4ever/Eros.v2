'' Launches Eros in the background with no console window.
'' Put a shortcut to this in shell:startup to run on boot.
Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw """ & WScript.ScriptFullName & """\..\run.py --wake --tray", 0, False
