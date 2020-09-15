import os
import math
from decimal import Decimal

import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm

import numpy as np
import pdb

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (haze, A, t, latent, filename, _) in enumerate(self.loader_train):
            haze, A, t, latent = self.prepare([haze, A, t, latent])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            A_iter1, t_iter1, J_iter1, A_iter2, t_iter2, J_iter2, A_iter3, t_iter3, J_iter3, A_iter4, t_iter4, J_iter4 = self.model(haze)
            loss = self.loss(A_iter1, A) + self.loss(t_iter1, t) + self.loss(J_iter1, latent) + \
                self.loss(A_iter2, A) + self.loss(t_iter2, t) + self.loss(J_iter2, latent) + \
                self.loss(A_iter3, A) + self.loss(t_iter3, t) + self.loss(J_iter3, latent) + \
                self.loss(A_iter4, A) + self.loss(t_iter4, t) + self.loss(J_iter4, latent)
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()
        # pdb.set_trace()

        self.feature_map_visualization = []
        def module_forward_hook(module, input, output):
            input_numpy = input[0].squeeze().cpu().numpy()
            output_numpy = output[0].squeeze().cpu().numpy()
            self.feature_map_visualization.append(input_numpy)
            self.feature_map_visualization.append(output_numpy)

        timer_test = utility.timer()
        with torch.no_grad():
            eval_acc = 0
            eval_acc_iter2 = 0
            eval_acc_iter3 = 0
            eval_acc_iter4 = 0
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (haze, latent, filename, _) in enumerate(tqdm_test):
                filename = filename[0]
                no_eval = (latent.nelement() == 1)
                if not no_eval:
                    haze, latent = self.prepare([haze, latent])
                else:
                    haze = self.prepare([haze])[0]

                # print(self.model)
                if not self.args.test_only:
                    A_iter1, t_iter1, J_iter1, A_iter2, t_iter2, J_iter2, A_iter3, t_iter3, J_iter3, A_iter4, t_iter4, J_iter4 = self.model(haze)
                else:
                    # handle = self.model.model.resnet50[0].layer4[2].relu.register_forward_hook(module_forward_hook)
                    A_iter1, t_iter1, J_iter1, A_iter2, t_iter2, J_iter2, A_iter3, t_iter3, J_iter3, A_iter4, t_iter4, J_iter4 = self.model(haze)
                    # handle.remove()
                
                A_iter1 = utility.quantize(A_iter1, self.args.rgb_range)
                t_iter1 = utility.quantize(t_iter1, self.args.rgb_range)
                J_iter1 = utility.quantize(J_iter1, self.args.rgb_range)
                A_iter2 = utility.quantize(A_iter2, self.args.rgb_range)
                t_iter2 = utility.quantize(t_iter2, self.args.rgb_range)
                J_iter2 = utility.quantize(J_iter2, self.args.rgb_range)
                A_iter3 = utility.quantize(A_iter3, self.args.rgb_range)
                t_iter3 = utility.quantize(t_iter3, self.args.rgb_range)
                J_iter3 = utility.quantize(J_iter3, self.args.rgb_range)
                A_iter4 = utility.quantize(A_iter4, self.args.rgb_range)
                t_iter4 = utility.quantize(t_iter4, self.args.rgb_range)
                J_iter4 = utility.quantize(J_iter4, self.args.rgb_range)

                save_list = [haze]
                if not no_eval:
                    eval_acc += utility.calc_psnr(
                        J_iter1, latent, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_acc_iter2 += utility.calc_psnr(
                        J_iter2, latent, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_acc_iter3 += utility.calc_psnr(
                        J_iter3, latent, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_acc_iter4 += utility.calc_psnr(
                        J_iter4, latent, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    save_list.extend([latent, A_iter1, t_iter1, J_iter1, \
                    A_iter2, t_iter2, J_iter2, A_iter3, t_iter3, J_iter3, A_iter4, t_iter4, J_iter4])

                if self.args.save_results:
                    self.ckp.save_results(filename, save_list)
            
            self.ckp.log[-1, 0] = eval_acc / len(self.loader_test)
            best = self.ckp.log.max(0)
            self.ckp.write_log(
                    '[{}]\tPSNR of iter1: {:.3f}, PSNR of iter2: {:.3f}, PSNR of iter3: {:.3f}, PSNR of iter4: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        self.ckp.log[-1, 0],
                        eval_acc_iter2/len(self.loader_test),
                        eval_acc_iter3/len(self.loader_test),
                        eval_acc_iter4/len(self.loader_test),
                        best[0][0],
                        best[1][0] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

