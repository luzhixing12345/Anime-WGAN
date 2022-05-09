
class a:
    
    def __init__(self):
        self.x = 1
        
    def __getitem__(self, key):
        return self.x + key
    
a.__getitem__ = lambda self,key: self.x+key*10
s = a()

print(s[2])