#SingleInstance, Force ; skips the dialog box and replaces the old instance automatically
#IfWinActive
Send {Up down}  ; Press down the up-arrow key.
Sleep 100  ; Keep it down for 100m second.
Send {Up up}  ; Release the up-arrow key.
#IfWinActive