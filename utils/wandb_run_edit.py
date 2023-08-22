import wandb
api = wandb.Api()

def updated_value(run_id, updated_value):
    run = api.run(f"soumyabanerjee/Privacy-Preverving-Ferderated-Learning{run_id}")
    run.config["key"] = updated_value
    run.update()