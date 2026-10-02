from config import ConfigParser

from Ray.ray import (
    RayOrchestrator,
)


class Application:

    def __init__(self):

        self.config = None
        self.orchestrator = None

    def run(self):

        self.config = (
            ConfigParser().parse()
        )

        self.orchestrator = (
            RayOrchestrator(
                self.config
            )
        )

        self.orchestrator.run()


if __name__ == "__main__":

    Application().run()