import torch
import torch.nn.functional as F
from ignite.engine import Engine, Events


def create_supervised_trainer(model, optimizer, loss_fn, device=None):
	if device is not None:
		model.to(device)

	def _update(engine, batch):
		model.train()
		optimizer.zero_grad()
		data, target = batch
		if device is not None:
			data = data.to(device)
			target = target.to(device)
		feat, out = model(data)
		loss = loss_fn(feat, out, target)
		loss.backward()
		optimizer.step()
		return loss.item()

	return Engine(_update)


def do_train_metric(opt, model, train_loader, test_loader, optimizer, loss_fn):
	trainer = create_supervised_trainer(model, optimizer, loss_fn, device=opt.device)
	# evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(), 'loss': Loss(loss_fn)}, device=opt.device)

	@trainer.on(Events.ITERATION_COMPLETED)
	def log_training_loss(engine):
		iter = (engine.state.iteration - 1) % len(train_loader) + 1

		if iter % opt.log_interval == 0:
			print('Loss:', engine.state.output)

	# @trainer.on(Events.EPOCH_COMPLETED)
	# def log_training_result(engine):
	# 	evaluator.run(train_loader)
	# 	metrics = evaluator.state.metrics
	# 	avg_accuracy = metrics['accuracy']
	# 	avg_loss = metrics['loss']
	# 	print('Training result: Epoch', engine.state.epoch, 'Accuracy:', avg_accuracy, 'Loss:', avg_loss)

	# @trainer.on(Events.EPOCH_COMPLETED)
	# def log_validation_results(engine):
	# 	evaluator.run(test_loader)
	# 	metrics = evaluator.state.metrics
	# 	avg_accuracy = metrics['accuracy']
	# 	avg_loss = metrics['loss']
	# 	print('Validation result: Epoch', engine.state.epoch, 'Accuracy:', avg_accuracy, 'Loss:', avg_loss)

	trainer.run(train_loader, max_epochs=opt.n_epochs)
