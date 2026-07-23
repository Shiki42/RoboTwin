import transformers


EXPECTED_TRANSFORMERS_VERSION = "4.53.2"


def check_whether_transformers_replace_is_installed_correctly():
    return transformers.__version__ == EXPECTED_TRANSFORMERS_VERSION
