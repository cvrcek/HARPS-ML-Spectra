from typing import Any


def update_config(config_dic: dict[str, Any]) -> dict[str, Any]:
    keys2remove_in_trainer = ['num_processes', 'gpus', 'auto_select_gpus', 'tpu_cores',
                              'ipus', 'track_grad_norm', 'resume_from_checkpoint', 'auto_lr_find',
                              'replace_sampler_ddp', 'auto_scale_batch_size', 'amp_backend',
                              'amp_level', 'move_metrics_to_cpu', 'multiple_trainloader_mode',]

    for key in keys2remove_in_trainer:
        if key in config_dic['trainer']:
            config_dic['trainer'].pop(key)
    config_dic['trainer']['strategy'] = 'auto'
    

    
    config_dic['data']['labels']['class_path'] = 'datasets.Labels.Labels'

    return config_dic