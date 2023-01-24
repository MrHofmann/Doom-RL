# Changes seconds to some nice string
def sec_to_str(secs):
    res = str(int(secs % 60)) + "s"
    mins = int(secs / 60)
    if mins > 0:
        res = str(int(mins % 60)) + "m " + res
        hours = int(mins / 60)
        if hours > 0:
            res = str(int(hours % 60)) + "h " + res
    return res

class AgentDebug:
    def __init__(self, debug_level):
        self.debug_level = debug_level

    def start(self, message, current_level):
        if self.debug_level >= current_level:
            for i in range(current_level - 1):
                print("", end="\t")
            print(message)

        return time()

    def end(self, message, current_level, start_time):
        if self.debug_level >= current_level:
            stop_time = time()
            duration = stop_time - start_time
            for i in range(current_level - 1):
                print("", end="\t")
            print(message + " " + str(duration))

doom_debug = AgentDebug(0)
