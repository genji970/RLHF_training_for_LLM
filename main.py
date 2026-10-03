from config import ConfigParser
from orchestrator import RayOrchestrator


class Application:
    def run(self):
        RayOrchestrator(ConfigParser().parse()).run()


if __name__ == "__main__":
    Application().run()
