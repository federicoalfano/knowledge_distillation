def distill_knowledge(teacher_model, student_model, train_dataset, fit_args):
  def custom_generator(train_dataset, t_model):
    for (x, y) in train_dataset:
      y_targets = teacher_model(x)
      yield (x, y_targets)

  s_history = student_model.fit(custom_generator(train_dataset, 
                                                 t_model=teacher_model),             
                      **fit_args)

  return s_history


def ban(teacher_model, n_students, build_model, 
        train_dataset, fit_args,
        checkpoints=False,
        compile_args={'optimizer':'adam', 
                      'loss': 'categorical_crossentropy', 
                      'metrics': ['accuracy']}):
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
    current_history = distill_knowledge(students[i-1], 
                                        students[i], train_dataset,
                                        fit_args)   
    history.append(current_history)
  return history, students
