

def check_attr(args,attr = 'attention_norm'):
    if not hasattr(args, attr):
        setattr(args, attr, False)