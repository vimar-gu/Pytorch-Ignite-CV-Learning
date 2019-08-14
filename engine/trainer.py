import os
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from utils.visualise import visualise

global best_accuracy
best_accuracy = 0


def do_train_normal(opt, model, train_loader, test_loader, optimizer, loss_fn):
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=opt.device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(loss_fn)}, device=opt.device)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % opt.log_interval == 0:
            print('Loss:', engine.state.output)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_result(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        print('Training result: Epoch', engine.state.epoch, 'Accuracy:', avg_accuracy, 'Loss:', avg_loss)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']

        global best_accuracy
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            torch.save(model.state_dict(), os.path.join(opt.output_dir, 'best_validation.pth'))

        print('Validation result: Epoch', engine.state.epoch, 'Accuracy:', avg_accuracy, 'Loss:', avg_loss)

    trainer.run(train_loader, max_epochs=opt.n_epochs)


def inference(opt, model, val_loader):
    evaluator = create_supervised_evaluator(model, device=opt.device)
    feats = []
    targets = []

    @evaluator.on(Events.ITERATION_COMPLETED)
    def cat_feats_targets(engine):
        feat, target = engine.state.output
        feats.append(feat)
        print(feats)
        targets.extend(target.cpu().numpy())

    evaluator.run(val_loader)
    visualise(opt, feats, targets)
