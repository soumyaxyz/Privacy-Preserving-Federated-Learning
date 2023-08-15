import wandb
api = wandb.Api()

run = api.run("soumyabanerjee/Privacy-Preverving-Ferderated-Learning/<run_id>")
run.config["key"] = updated_value
run.update()