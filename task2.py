# https://github.com/Lightning-AI-Dev/pyconf-hyd-2022
"""Create a LightningFlow

1. LightningFlow contains the work we created `TaskOne`
2. In the run method execute the TaskOne work
3. Count how many times 'hello world' is printed
"""
import lightning as L
import time
class TaskOne(L.LightningWork):
    def __init__(self):
        super().__init__(parallel=True, cloud_compute=L.CloudCompute("gpu"))
        self.demo = "Hello World"

    def run(self):
        print(self.demo)
        time.sleep(5)
        print("awake")


class TaskManager(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.task1_work = TaskOne()
    
    def run(self):
        print('hello')
        self.task1_work.run()
        # self.task1_work.stop()
        if self.task1_work.has_stopped:
            print("stopped")

component = TaskManager()
app = L.LightningApp(component)