import argparse
import os
class NMTArgument:
    def __init__(self):
        data = {}
        parser = self.get_args()
        args = parser.parse_args()

        data.update(vars(args))
        self.data = data
        self.set_savename()
        self.__dict__ = data

    def get_args(self):
        parser = argparse.ArgumentParser(description='NMT 프로젝트 ver 1.0')  # 아래와 같은 형태로 .py 파일에 접근해 실행하는 옵션 달아주는 모듈.

        parser.add_argument("--root", type=str, default="data")  # dataset 이 들어가는 path
        parser.add_argument("--n_epoch", default=10, type=int)
        parser.add_argument("--seed", default=777, type=int)    # seed 고정을 해야 재학습시 동일한 결과 반환됨 동일한 환경서 성능 비교해야하므로
        parser.add_argument("--per_gpu_train_batch_size", default=64, type=int)
        parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int)
        parser.add_argument("--gradient_accumulation_step", default=1, type=int)
        parser.add_argument("--seq_len", default=512, type=int)  # 학습 과정에서 제일 길게 들어가는 input  < >
        parser.add_argument("--warmup_step", default=0, type=int)   # linear lr scheduler < lr 끝까지 고정하는 방식으로 짬 >
        parser.add_argument("--decay_step", default=20000, type=int)    # lr 스케쥴러랑 비슷한 거래
        parser.add_argument("--clip_norm", default=0.25, type=float)
        parser.add_argument("--replc", default=0.25, type=float)

        parser.add_argument("--lr", default=1e-5, type=float)   # lr

        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--do_train", action="store_true")
        parser.add_argument("--do_eval", action="store_true")
        parser.add_argument("--evaluate_during_training", action="store_true")
        parser.add_argument("--do_test", action="store_true")
        parser.add_argument("--checkpoint_dir", default="checkpoints", type=str)    # checkpoint 디렉토리 먼저 지정하고, 학습해야, cp 기반해서 학습된 weight 저장된다.
        parser.add_argument("--max_train_steps", default=None, type=int)
        parser.add_argument("--lr_scheduler_type", default="linear", help="The scheduler type to use.",
                            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                     "constant_with_warmup"])

        parser.add_argument("--num_warmup_steps", type=int, default=0,
                            help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--replace_vocab", action="store_true")

        return parser

    def set_savename(self):

        self.data["savename"] = os.path.join(self.data["checkpoint_dir"], f"vanila")    # dir 이름, checkpoint 저장 디렉토리, epoch 별로 저장할 수 있다.

