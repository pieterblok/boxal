The following settings can probably stay unchanged: <br/> <br/>

| Setting        			| Description        													|
| --------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| duplicate_initial_model_and_data	| Experimental mode: set this to **True** when you want to duplicate a previously trained model and dataset with a new settings-file. Default **False**	|
| initial_train_file			| When **duplicate_initial_model_and_data** is set to True, then specify the txt-file with the initial dataset.		|
| transfer_learning_on_previous_models	| Whether to use the weight-files from the previous training iterations for transfer-learning				|
| warmup_iterations			| The number of warmup-iterations that can be used to stabilize the training process 			 		|
| train_iterations_base			| The number of training iterations to initialize the training (this number of training iterations is used when the total number of training images is below the value of **step_image_number**)								 			 		|
| train_iterations_step_size		| When the number of training images exceeds the **step_image_number**, then this number of iterations is added to the **train_iterations_base**																	|
| step_image_number			| The number of training images to increase the number of iterations specified in **train_iterations_step_size**	|
| eval_period				| A multitude of training iterations to perform the evaluation on the validation set					|
| checkpoint_period			| The number of training iterations to store the training weights (use **-1** to disable intermediate checkpoints)	|
| weight_decay	 			| The weight-decay value to train the object detector									|
| learning_policy 			| The learning-policy to train the object detector									|
| step_ratios				| When the training iterations reach this ratio, then the learning rate is automatically lowered by a fraction of 0.1 	|
| gamma		 			| The gamma-value to train the object detector										|
| train_batch_size 			| The image batch-size to train the object detector									|
| num_workers	 			| The number of workers to train the object detector									|
| train_sampler	 			| The data-sampler to train the object detector. Use **"RepeatFactorTrainingSampler"**, when there is class-imbalance	|
| minority_classes 			| Only when the **"RepeatFactorTrainingSampler"** is used: specify the minority-classes that have to be repeated	|
| repeat_factor_smallest_class		| Only when the **"RepeatFactorTrainingSampler"** is used: specify the repeat-factor of the smallest class (use a value higher than 1.0 to repeat the minority classes)																	|
| experiment_name			| Specify the name of your experiment											|
| strategy				| Use **'uncertainty'** to select the most uncertain images for the active learning. Other options are **'random'** and **'certainty'** |
| mode					| Uncertainty sampling method. Use **'mean'** when you want to sample the most uncertain images, use **'min'** when you want to sample the most uncertain instances																	|
| equal_pool_size			| When **True** this will sample the same **pool_size** for every sampling iteration. When **False**, an unequal **pool_size** will be sampled for the specified number of loops															|
| dropout_probability			| Specify the dropout probability between 0.1 and 0.9. Our experiments indicated that **0.25** and **0.50** are good values	|
| mcd_iterations			| The number of Monte-Carlo iterations to calculate the uncertainty of the image. When this number is increased, the uncertainty metric will be more consistent. When this number is decreased, the sampling will be faster. The value **10** is a good compromise between consistency and speed	|
| iou_thres				| Intersection of Union threshold to cluster the different instance segmentations into observations for the uncertainty calculation																			|
<br/>
