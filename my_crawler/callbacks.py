from context import Context


def db_update_callback(status):
    pass

def db_update_tag(status):
    with Context() as ctx:
        assert ctx.db_utils is not None
        # ctx.db_utils.exists()