from torch.utils import data as data_utils

from models import model_factory


def add_general_flags(parser):
    parser.add_argument('--save', default='checkpoints',
                        help="Path to the directory to save logs and "
                        "checkpoints.")
    parser.add_argument('--gpus', '--gpu', nargs='+', default=[0], type=int,
                        help="The GPU(s) on which the model should run. The "
                        "first GPU will be the main one.")
    parser.add_argument('--cpu', action='store_const', const=[],
                        dest='gpus', help="If set, no gpus will be used.")


def add_dataset_flags(parser):
    parser.add_argument('--imagenet', required=True, help="Path to ImageNet's "
                        "root directory holding 'train/' and 'val/' "
                        "directories.")
    parser.add_argument('--batch-size', default=256, help="Batch size to use "
                        "distributed over all GPUs.", type=int)
    parser.add_argument('--num-workers', '-j', default=40, help="Number of "
                        "data loading processes to use for loading data and "
                        "transforming.", type=int)
    parser.add_argument('--image-size', default=224, help="image size to train "
                        "input image size.", type=int)


def add_model_flags(parser):
    parser.add_argument('--model', required=True, help="The model architecture "
                        "name.")
    parser.add_argument('--student-state-file', default=None, help="Path to student model"
                        "state file to initialize the student model.")
    parser.add_argument('--start-epoch', default=1, help="manual epoch number "
                        "useful on restarts.", type=int)
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')


def add_teacher_flags(parser):
    parser.add_argument('--teacher-model', default="vgg19_bn,resnet50", help="The "
                        "model that will generate soft labels per crop.",
                        )
    parser.add_argument('--teacher-state-file', default=None,
                        help="Path to teacher model state file.")


def add_training_flags(parser):
    parser.add_argument('--lr-regime', default=None, nargs='+', type=float,
                        help="If set, it will override the default learning "
                        "rate regime of the model. Learning rate passed must "
                        "be as list of [start, end, lr, ...].")
    parser.add_argument('--d_lr', default=1e-4,  type=float,
                        help="The learning rate for discriminator training")
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="The momentum of the optimization.")
    parser.add_argument('--weight-decay', default=0, type=float,
                        help="The weight decay of the optimization.")
