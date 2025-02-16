from fuglu.shared import ScannerPlugin, DUNNO, Suspect

class DemoPlugin(ScannerPlugin):
    """Demo plugin"""

    def __init__(self, config, section=None):
        ScannerPlugin.__init__(self, config, section)
        print("initialize demo plugin")

    def examine(self, suspect: Suspect) -> int:
        print(f"demo plugin examine suspeect={suspect}")
        print(f"msg source = {suspect.get_source()}")
        self._logger().info('hello world from DemoPlugin')
        return DUNNO

    def lint(self):
        allok=(self.check_config())
        return allok

