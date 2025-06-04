import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        # self.save_path = config["save_path"]
        # self.predict_path = config["predict_path"]

        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]
        self.model_learning_rate = config["model_learning_rate"]
        self.bert_learning_rate = config["bert_learning_rate"]
        self.encoder_path = config["encoder_path"]
        self.train_datasets_path = config["train_datasets_path"]
        self.dev_datasets_path = config["dev_datasets_path"]
        self.test_datasets_path = config["test_datasets_path"]
        self.dropout = config["dropout"]
        self.ent_type_size = config["ent_type_size"]
        self.linear_size = config["linear_size"]
        self.biaffine_size = config["biaffine_size"]
        self.robust_weight = config["robust_weight"]
        self.extra_loss_weight = config["extra_loss_weight"]
        self.seed = config["seed"]
        self.label2id = config["label2id"]
        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    def __repr__(self):
        return "{}".format(self.__dict__.items())







