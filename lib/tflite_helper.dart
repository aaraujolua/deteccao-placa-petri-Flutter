// ignore_for_file: avoid_print

import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

class TFLiteHelper {
  late Interpreter _interpreter;

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/best_yolov5_placas_quantized.tflite',
      );
      print('✅ Modelo TFLite carregado!');
      // Impressão robusta do número de tensores (funciona em 0.9.x e 0.11.x)
      try {
        final dyn = _interpreter as dynamic;
        final inCount =
            (dyn.inputTensors != null) ? dyn.inputTensors.length : dyn.getInputTensors().length;
        final outCount =
            (dyn.outputTensors != null) ? dyn.outputTensors.length : dyn.getOutputTensors().length;
        print('Entradas: $inCount | Saídas: $outCount');
      } catch (_) {/* opcional ignorar */}
    } catch (e) {
      print('❌ Erro ao carregar modelo: $e');
      rethrow;
    }
  }

  Future<List<dynamic>> runInference(File imageFile) async {
    try {
      final rawBytes = await imageFile.readAsBytes();
      final decoded = img.decodeImage(rawBytes);
      if (decoded == null) {
        print('❌ Não foi possível decodificar a imagem.');
        return [];
      }

      // TODO: ajuste para o tamanho real do seu modelo (ex.: 640x640/320x320)
      const inputW = 224;
      const inputH = 224;

      final resized = img.copyResize(
        decoded,
        width: inputW,
        height: inputH,
        interpolation: img.Interpolation.linear,
      );

      // Para modelo FLOAT32 normalizado [0..1]
      final input = List.generate(
        1,
        (_) => List.generate(
          inputH,
          (y) => List.generate(
            inputW,
            (x) {
              final px = resized.getPixel(x, y); // img.Pixel (image 4.x)
              final r = px.r.toDouble() / 255.0;
              final g = px.g.toDouble() / 255.0;
              final b = px.b.toDouble() / 255.0;
              return [r, g, b];
            },
          ),
        ),
      );

      // Buffer de saída placeholder — ajuste pro shape do seu YOLOv5 TFLite
      final output = List.filled(1 * 25200 * 7, 0.0).reshape([1, 25200, 7]);

      _interpreter.run(input, output);
      print('✅ Inferência executada!');
      return output;
    } catch (e) {
      print('❌ Erro na inferência: $e');
      return [];
    }
  }

  Interpreter get interpreter => _interpreter;
}
