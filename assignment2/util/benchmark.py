import time

class Timer:
    """Timer is a simple class for timing code.
    """
    def __init__(self):
        self.reset()

    def begin(self, start_id):
        """Begin timing a piece of code.

        Args:
            start_id: a unique identifier for where to start the timer
        """
        self.start[start_id] = time.time()

    def getElapsed(self, start_id):
        """Get the elapsed time since calling begin(start_id).

        Args:
            start_id: the unique identifier for when timing was started

        Returns:
            elapsed time in seconds
        """
        if start_id not in self.start:
            raise ValueError("start_id not found")
        end = time.time()
        return end - self.start[start_id]

    def reset(self):
        """Reset by clearing all start_id's.
        """
        self.start = {}
