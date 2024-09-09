import torch


class data_prefetcher(object):
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.preload()

    def preload(self):
        try:
            self.x, self.y = next(self.loader)
        except StopIteration:
            self.x = None
            self.y = None
            return
        with torch.cuda.stream(self.stream):
            self.x = self.x.float().to(self.device)
            self.y = self.y.float().to(self.device)

    def next(self):
        torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
        x = self.x
        y = self.y
        self.preload()
        return x, y

    def __len__(self):
        return len(self.loader)
