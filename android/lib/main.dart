import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:tflite/tflite.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  // This widget is the root of your application.

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Bird Classifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  @override
  _PredictionPageState createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  File _pickedImage;

  List _outputs;

  void initState() {
    super.initState();

    loadModel().then((value) {});
  }

  loadModel() async {
    await Tflite.loadModel(
        model: "assets/model.tflite", labels: "assets/labels.txt");
  }

  getLabelName(String string) {
    int index = string.indexOf(" ");
    return string.substring(index);
  }

  classifyImage(File image) async {
    var output = await Tflite.runModelOnImage(
        path: image.path, numResults: 1, imageMean: 0, imageStd: 255);
    setState(() {
      _outputs = output;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Bird Classifier'),
      ),
      body: ListView(
        children: <Widget>[
          const SizedBox(height: 15.0),
          Center(
              child: SizedBox(
            height: 240,
            child: Container(
                child: (_pickedImage == null
                    ? Container()
                    : Image.file(_pickedImage))),
          )),
          const SizedBox(height: 10.0),
          Container(
            height: 50,
            margin: EdgeInsets.symmetric(horizontal: 50),
            child: RaisedButton(
              color: Colors.blue,
              textColor: Colors.white,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(18.0),
              ),
              child: Text("Pick an Image"),
              onPressed: () {
                _showPickOptionDialog(context);
              },
            ),
          ),
          SizedBox(
            height: 50,
          ),
          Column(
            children: [
              _outputs != null
                  ? Text(
                      getLabelName(_outputs[0]['label']),
                      style: TextStyle(
                        fontSize: 30,
                        fontWeight: FontWeight.bold,
                      ),
                    )
                  : Text(
                      "Classification Waiting",
                      style: TextStyle(fontSize: 15),
                    ),
            ],
          )
        ],
      ),
    );
  }

  _loadPicker(ImageSource source) async {
    File picked = await ImagePicker.pickImage(source: source);
    if (picked != null) {
      setState(() {
        _cropImage(picked);
      });
    }
    Navigator.pop(context);
  }

  _cropImage(File picked) async {
    File cropped = await ImageCropper.cropImage(
        androidUiSettings: AndroidUiSettings(
          statusBarColor: Colors.blue,
          toolbarColor: Colors.blue,
          toolbarTitle: "Crop Image",
          toolbarWidgetColor: Colors.white,
          activeControlsWidgetColor: Colors.blue
        ),
        sourcePath: picked.path,
        aspectRatioPresets: [
          CropAspectRatioPreset.square,
          CropAspectRatioPreset.ratio3x2,
          CropAspectRatioPreset.original,
          CropAspectRatioPreset.ratio4x3,
          CropAspectRatioPreset.ratio16x9
        ]);
    if (cropped != null) {
      setState(() {
        _pickedImage = cropped;

        classifyImage(_pickedImage);
      });
    }
  }

  void _showPickOptionDialog(BuildContext context) {
    showDialog(
        context: context,
        builder: (context) => AlertDialog(
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: <Widget>[
                  ListTile(
                    title: Text("Pick from Gallery"),
                    onTap: () {
                      _loadPicker(ImageSource.gallery);
                    },
                  ),
                  ListTile(
                    title: Text("Take a picture"),
                    onTap: () {
                      _loadPicker(ImageSource.camera);
                    },
                  )
                ],
              ),
            ));
  }
}
