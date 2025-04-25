import mouse
import time

# mouse.press('left')
# mouse.move(100, 100, absolute=False, duration=0.2)
# mouse.release('left')

# while True:
#     print(mouse.get_position())

# every 5 second, move the mouse to the position (-1849, 49) and click the left button
# while True:
#     mouse.move(-1849, 49)
#     mouse.click('left')
#     print("Page refreshed!")
#     mouse.move(100, 100)
#     time.sleep(10)

time.sleep(5)
mouse.move(760, 540)
time.sleep(10)
mouse.move(1160, 540)

# mouse.click('right')