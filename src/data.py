print("Hello world");

class MyDataset(Dataset):
    def _init_(self,file):
        self.data= ...

    def _getitem_(self,index):
        return self.data[index]
    def _len_(self):
        return len(self.data)