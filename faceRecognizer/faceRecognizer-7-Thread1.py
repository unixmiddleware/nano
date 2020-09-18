from threading import Thread
import time

def BigBox(color,x):
    while True:
        print(color,'BigBox is Open')
        time.sleep(5)
        print(color,'BigBox is Closed')
        time.sleep(5)

def SmallBox(color,x):
    while True:
        print(color,'SmallBox is Open')
        time.sleep(1)
        print(color,'SmallBox is Closed')
        time.sleep(1)

SmallBoxThread=Thread(target=SmallBox,args=['red',4])
BigBoxThread=Thread(target=BigBox,args=['blue',5])

SmallBoxThread.daemon = True
BigBoxThread.daemon = True

SmallBoxThread.start()
BigBoxThread.start()

time.sleep(20)
