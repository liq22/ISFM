

def extract_task_name(args_d):
    if isinstance(args['dataset']['task'],str):
        task = args['dataset']['task']
    elif isinstance(args['dataset']['task'],dict):
        # 提取所有的key值
        task_keys = list(args['dataset']['task'].keys())
        # 如果键数量大于3个，则只使用前三个，后续用数字表示
        if len(task_keys) > 3:
            task = '_'.join(task_keys[:3]) + f'_plus{len(task_keys)-3}'
        else:
            task = '_'.join(task_keys)
    return task