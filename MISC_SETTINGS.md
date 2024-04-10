The following settings can probably stay unchanged: <br/> <br/>

| Setting        			| Description        													|
| --------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
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
<br/>
