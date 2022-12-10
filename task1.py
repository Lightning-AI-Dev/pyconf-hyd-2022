"""
Create a LightningWork 
1. store hello world in self.demo
2. Run the LightningWork and count how many times "hello world" is printed.

"""
import lightning as L

class TaskOne(L.LightningWork):
    def __init__(self):
        super().__init__()
        self.demo = "Hello World"

    def run(self):
        print(self.demo)

component = TaskOne()
app = L.LightningApp(component)
