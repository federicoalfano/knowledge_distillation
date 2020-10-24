def distil_knowledge(teacher_model, student_model, train_dataset, 
                      fit_args, ground_truth_weight=None):
  """
The method aims to match the teacher predictions on a student, optionally it can
adjust its prediction with the aid of a given weight
  Reference:
    - [Distilling the Knowledge in a Neural Network](
        https://arxiv.org/abs/1503.02531 )
  Arguments:
    teacher_model: The model from which to distil knowledge.
    student_model: The target model that have to match its teacher's predictions
    train_dataset: That's the dataset used to train the teacher generally a generator
    fit_args: dictionary containing the arguments to pass to the student during the fitting, it 
    doesn't need the training_dataset
    ground_truth_weight: a decimal between 0 and 1, it makes the ground_truth influencing 
    the prediction to match

  Returns:
    history of the student model
  Raises:
    ValueError: in case of an invalid ground_truth_weight
  """

  if (ground_truth_weight is not None and (ground_truth_weight > 1 or ground_truth_weight < 0)):
    raise ValueError("Please check the ground truth weight")
  def custom_generator(train_dataset, t_model, ground_truth_weight):
    for (x, y) in train_dataset:
      y_targets = teacher_model(x)
      if ground_truth_weight is not None:
        y_targets = (1-ground_truth_weight)*y_targets+(ground_truth_weight)*y
      yield (x, y_targets)

  s_history = student_model.fit(custom_generator(train_dataset, 
                                                 t_model=teacher_model,
                                                 ground_truth_weight=ground_truth_weight),             
                                                 **fit_args)

  return s_history


def ban(teacher_model, n_students, build_model, 
        train_dataset, fit_args,
        checkpoints=False,
        ground_truth_weight=None,
        compile_args={'optimizer':'adam', 
                      'loss': 'categorical_crossentropy', 
                      'metrics': ['accuracy']}):

  """
  The method trains some generations of students distilling the knowledge from the previous one,
  playing with build_model and compile args, allows you to perform different kinds of knowledge
  distillation
  
  Reference:
    - [Born Again Neural Networks](
        https://arxiv.org/abs/1805.04770)
  Arguments:
    teacher_model: The first model from which to distil knowledge, generally trained on dataset
    n_students: The number of students' generation to train
    build_model: the function called to build every student 
    train_dataset: That's the reference dataset used to train the teacher generally a generator
    fit_args: dictionary containing the arguments to pass to the student during the fitting, it 
    doesn't need the training_dataset
    check_points: if True save the best weights of student in the current directory with name
    "ban-n" and n is the generation number
    ground_truth_weight: a decimal between 0 and 1, it makes the ground_truth influencing 
    the prediction to match
    compile_args: a dictionary with arguments used to compile the students

  Returns:
    A tuple containing a list of histories and a students. The list of students has at first
    position the teacher model just to simplify the evaluation, if you don't need it, simply pop it
  

  """
  students = [build_model() for i in range(n_students)]
  students.insert(0, teacher_model)
  for student in students:
    student.compile(**compile_args)
  history = []


  for i in range(1, len(students)):
    print("Training BAN-{}".format(i))
    current_callbacks = fit_args['callbacks']
    if checkpoints:
      current_callbacks.append(tf.keras.callbacks.ModelCheckpoint(f"student.{i}.h5"))
      fit_args['callbacks'] = current_callbacks
    current_history = distil_knowledge(students[i-1], 
                                        students[i], train_dataset,
                                        fit_args,
                                        ground_truth_weight=ground_truth_weight)   
    history.append(current_history)
  return history, students

