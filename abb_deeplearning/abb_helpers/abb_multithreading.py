import threading
import time

"""
class ABBThread (threading.Thread):
    def __init__(self, threadID, threadName='ABBthread', function_obj, *args):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.threadName = threadName
        self.task = function_obj
        self.arguments = args

    def run(self):
        print("Start " + self.threadName)
        self.task(self.arguments)
        print("Ending " + self.threadName)
"""