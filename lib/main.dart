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
  final ImagePicker _picker = ImagePicker();
  final tfliteHelper = TFLiteHelper();

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    await tfliteHelper.loadModel();
  }

  Future<void> _openCamera() async {
    final pickedFile = await _picker.pickImage(source: ImageSource.camera);
    if (pickedFile != null) {
      setState(() => _imageFile = File(pickedFile.path));
      await _runModel(_imageFile!);
    }
  }

  bool _isPicking = false;

  Future<void> _pickFromGallery() async {
    if (_isPicking) return; // bloqueia reentradas
    _isPicking = true;
    try {
      final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
      if (pickedFile != null) {
        setState(() => _imageFile = File(pickedFile.path));
        await _runModel(_imageFile!);
      }
    } finally {
      _isPicking = false;
    }
  }

  Future<void> _runModel(File image) async {
    print('🧠 Rodando inferência...');
    final results = await tfliteHelper.runInference(image);

    if (results.isEmpty) {
      print('⚠️ Nenhum objeto detectado.');
    } else {
      print('📊 ${results.length} objetos detectados:');
      for (final det in results.take(3)) {
        print(
          '→ Classe ${det['classId']} | Confiança ${(det['confidence'] * 100).toStringAsFixed(1)}%',
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.purple.shade50,
      appBar: AppBar(
        title: const Text("Detecção de Placas"),
        backgroundColor: Colors.purple.shade300,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              ElevatedButton.icon(
                onPressed: _openCamera,
                icon: const Icon(Icons.camera_alt),
                label: const Text("Abrir Câmera"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.purple.shade300,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                onPressed: _pickFromGallery,
                icon: const Icon(Icons.photo_library),
                label: const Text("Selecionar da Galeria"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple.shade200,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(20),
                  ),
                ),
              ),
              const SizedBox(height: 20),
              _imageFile == null
                  ? const Text("Nenhuma imagem processada ainda.")
                  : ClipRRect(
                      borderRadius: BorderRadius.circular(15),
                      child: Image.file(_imageFile!, height: 300),
                    ),
            ],
          ),
        ),
      ),
    );
  }
}
