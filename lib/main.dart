import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'tflite_helper.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final tflite = TFLiteHelper();
  await tflite.loadModel();

  runApp(MyApp(tflite: tflite));
}

class MyApp extends StatelessWidget {
  final TFLiteHelper tflite;

  const MyApp({super.key, required this.tflite});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Detecção de Placas de Petri',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: HomePage(tflite: tflite),
    );
  }
}

class HomePage extends StatefulWidget {
  final TFLiteHelper tflite;

  const HomePage({super.key, required this.tflite});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File? _imageFile;
  bool _loading = false;
  String _status = "Nenhuma imagem processada ainda.";

  Future<void> _openCamera() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.camera);

    if (pickedFile != null) {
      setState(() {
        _imageFile = File(pickedFile.path);
        _status = "Processando...";
        _loading = true;
      });

      final result = await widget.tflite.runInference(_imageFile!);

      setState(() {
        _status = "Inferência concluída! Resultado: ${result.length} valores.";
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Detecção de Placas")),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              onPressed: _loading ? null : _openCamera,
              icon: const Icon(Icons.camera_alt),
              label: const Text("Abrir Câmera"),
            ),
            const SizedBox(height: 20),
            if (_loading) const CircularProgressIndicator(),
            if (_imageFile != null) Image.file(_imageFile!, height: 250),
            const SizedBox(height: 20),
            Text(_status, textAlign: TextAlign.center),
          ],
        ),
      ),
    );
  }
}
