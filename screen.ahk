#SingleInstance, Force ; skips the dialog box and replaces the old instance automatically

#include, Gdip.ahk
#persistent

raster:=0x40000000 + 0x00CC0020 ;to capture layered windows too

SetTimer, take_snapshot,420000 ;7*60*1000 for 7min
return

take_snapshot:
file=%A_Now%screen.png
screenshot(file,raster)
return


Screenshot(outfile,raster)
{
    pToken := Gdip_Startup()

    screen=0|0|%A_ScreenWidth%|%A_ScreenHeight%
    pBitmap := Gdip_BitmapFromScreen(screen,raster)

    Gdip_SaveBitmapToFile(pBitmap, outfile, 100)
    Gdip_DisposeImage(pBitmap)
    Gdip_Shutdown(pToken)
}