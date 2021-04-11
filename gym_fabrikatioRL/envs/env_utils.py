from os.path import join, exists
from os import makedirs


# <editor-fold desc="Exceptions">
class UndefinedSimulationModeError(Exception):
    def __init__(self):
        super().__init__('This simulation only accept modes '
                         '0 - scheduling decisions -, '
                         '1 - regular routing decisions - or '
                         '2 - routing decisions from behind a broken machine')


class MultipleParameterSetsError(Exception):
    def __init__(self):
        super().__init__('Cannot instantiate more than one simulation parameter'
                         ' sets! The class "EnvParameters" is a singleton."')


class InvalidHeuristicName(Exception):
    def __init__(self):
        super().__init__('Cannot instantiate ComparableAction: chosen heuristic'
                         ' name is unavailable.')


class InvalidStepNumber(Exception):
    def __init__(self):
        super().__init__(
            'The number of decisions exceeds the maximum allowed for this '
            'simulation run. Possible endless loop.')


class UndefinedInputType(Exception):
    def __init__(self, input_type, input_desc):
        super().__init__(f'The input type {input_type} is undefined for '
                         f'{input_desc}.')


class UndefinedOptimizerConfiguration(Exception):
    def __init__(self):
        err_string = f'The optimizer combination is not supported.'
        super().__init__(err_string)


class UndefinedOptimizerTargetMode(Exception):
    def __init__(self):
        err_string = (f'The optimizer target mode is not supported. Optimizers '
                      f'support "transport" and "sequencing" modes only')
        super().__init__(err_string)


class IllegalAction(Exception):
    def __init__(self):
        err_string = ()
        super().__init__(err_string)
# </editor-fold>


# <editor-fold desc="Utility Functions">
def create_folders(path):
    """
    Switches between '/' (POSIX) and '\'(windows) separated paths, depending on
    the current platform all non existing folders.

    :param path: A '/' separated *relative* path; the last entry is considered
        to be the file name and won't get created.
    :return: The platform specific file path.
    """
    segments = path.split('/')
    if not bool(segments):
        return path
    path_dir = join(*segments[:-1])
    file = segments[-1]
    if not exists(path_dir):
        makedirs(path_dir)
    return join(path_dir, file)
# </editor-fold>
