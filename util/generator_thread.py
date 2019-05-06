from threading import Thread

from Queue import Queue

class GeneratorThread:
    def __init__(self, iterators, max_storage=10):
        self.iterators = iterators
        self.max_storage = max_storage
        self.queue = Queue(maxsize=max_storage)
        for iterator in iterators:
            self.thread = Thread(name="GeneratorThread", target=lambda: self.run(iterator))
            self.thread.daemon = True
            self.thread.start()

    def run(self, iterator):
        while True:
            self.queue.put(next(iterator))
            if self.queue.qsize() < 5:
                #print("getting bottlenecked, max={}".format(self.max_storage))
                pass

    def size(self):
        return self.queue.qsize()

    def generator(self):
        def gen(_q):
            while True:
                yield _q.get(True, None)

        return gen(self.queue)

    def get_iterator(self):
        return self.generator()
