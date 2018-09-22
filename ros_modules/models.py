class Gesture(object):

    def __init__(self, start, end, data):
        self.start = start
        self.end = end
        self.data = data

    def get_array(self):
        return self.data

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end
