from .session import Session
from .utils.structures import traverse


class Simulation():

    def __init__(self, params):
        simulation_tree = params['children']['simulation']
        sessions_tree = params['children']['sessions']
        self.sessions_order = simulation_tree['sessions_order']
        self.sessions = {
            session_name: Session(session_params)
            for (session_name, session_params)
            in traverse(sessions_tree)
        }


    def run(self):
        for session in self.sessions_order:
            print(f'Run session `{session}`:')
            self.sessions[session].run()
