import os
import web
import cv2
import utils.game_manager as gm

urls = (
    '/', 'index'
)

DIR_PATH = 'D:\\map_data\\'
IMAGE_WIDTH           = 320
IMAGE_HEIGHT          = 240
IMAGE_CHANNEL         = 1 

class index:
    def POST(self):
    	screenshot = gm.grab_screen(region=(0, 40 , 1910, 1050))
    	data = str(web.data())
    	data = data.split("&")
    	screen_part = data[0]
    	action_part = data[1]
    	game_part = data[2]
    	screen_name = screen_part.split("=")[1]
    	action = action_part.split("=")[1]
    	game_id =  game_part.split("=")[1]
    	PATH = DIR_PATH + game_id+"\\"
    	if not os.path.exists(PATH):
    		os.makedirs(PATH)
    	screen = cv2.resize(screenshot, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    	screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY )
    	print(PATH+screen_name)
    	cv2.imwrite(PATH+screen_name, screen)
    	return True

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()