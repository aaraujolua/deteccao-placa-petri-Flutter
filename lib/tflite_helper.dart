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
        final inCount = (dyn.inputTensors != null)
            ? dyn.inputTensors.length
            : dyn.getInputTensors().length;
        final outCount = (dyn.outputTensors != null)
            ? dyn.outputTensors.length
            : dyn.getOutputTensors().length;
        print('Entradas: $inCount | Saídas: $outCount');
      } catch (_) {
        /* opcional ignorar */
      }
      // 🔍 Debug: imprime info do modelo
      final inputTensors = _interpreter.getInputTensors();
      final outputTensors = _interpreter.getOutputTensors();
      print('📥 Input tensor shape: ${inputTensors.first.shape}');
      print('📤 Output tensor shape: ${outputTensors.first.shape}');
    } catch (e) {
      print('❌ Erro ao carregar modelo: $e');
    }
  }

  // Future<List<Map<String, dynamic>>> runInference(File imageFile) async {
  //   try {
  //     print('🧠 Rodando inferência...');

  //     // Lê bytes e decodifica a imagem
  //     final rawBytes = await imageFile.readAsBytes();
  //     final decoded = img.decodeImage(rawBytes);
  //     if (decoded == null) {
  //       print('❌ Não foi possível decodificar a imagem.');
  //       return [];
  //     }

  //     // Tamanho esperado pelo modelo (ajuste se necessário)
  //     const inputW = 640;
  //     const inputH = 640;

  //     final resized = img.copyResize(
  //       decoded,
  //       width: inputW,
  //       height: inputH,
  //       interpolation: img.Interpolation.linear,
  //     );

  //     // Normaliza os valores RGB entre [0,1]
  //     final input = List.generate(
  //       1,
  //       (_) => List.generate(
  //         inputH,
  //         (y) => List.generate(inputW, (x) {
  //           final px = resized.getPixel(x, y);
  //           final r = px.r.toDouble() / 255.0;
  //           final g = px.g.toDouble() / 255.0;
  //           final b = px.b.toDouble() / 255.0;
  //           return [r, g, b];
  //         }),
  //       ),
  //     );

  //     // Cria o buffer de saída correto para o seu modelo (6 valores)
  //     final output = List.filled(1 * 25200 * 6, 0.0).reshape([1, 25200, 6]);

  //     // Executa o modelo
  //     _interpreter.run(input, output);
  //     print('✅ Inferência executada!');

  //     // Interpreta as detecções
  //     final detections = <Map<String, dynamic>>[];
  //     final results = output[0] as List;

  //     for (var obj in results) {
  //       final confidence = obj[4];
  //       if (confidence > 0.25) {
  //         final x = obj[0];
  //         final y = obj[1];
  //         final w = obj[2];
  //         final h = obj[3];
  //         final classId = obj[5].toInt();

  //         detections.add({
  //           'x': x,
  //           'y': y,
  //           'w': w,
  //           'h': h,
  //           'confidence': confidence,
  //           'classId': classId,
  //         });
  //       }
  //     }

  //     if (detections.isEmpty) {
  //       print('⚠️ Nenhum objeto detectado.');
  //     } else {
  //       print('📊 ${detections.length} objetos detectados:');
  //       for (var d in detections) {
  //         print(
  //           '→ Classe ${d['classId']} | Confiança ${(d['confidence'] * 100).toStringAsFixed(1)}%',
  //         );
  //       }
  //     }

  //     return detections;
  //   } catch (e) {
  //     print('❌ Erro na inferência: $e');
  //     return [];
  //   }
  // }

  Future<List<Map<String, dynamic>>> runInference(File imageFile) async {
    try {
      print('🧠 Rodando inferência...');

      // 1) Decodifica imagem original (guardaremos width/height originais p/ reescalar as caixas)
      final rawBytes = await imageFile.readAsBytes();
      final decoded = img.decodeImage(rawBytes);
      if (decoded == null) {
        print('❌ Não foi possível decodificar a imagem.');
        return [];
      }
      final origW = decoded.width.toDouble();
      final origH = decoded.height.toDouble();

      // 2) Redimensiona para o tamanho de entrada do seu modelo
      const inputW = 640;
      const inputH = 640;
      final resized = img.copyResize(
        decoded,
        width: inputW,
        height: inputH,
        interpolation: img.Interpolation.linear,
      );

      // 3) Prepara tensor [1, H, W, 3] float32 normalizado
      final input = List.generate(
        1,
        (_) => List.generate(
          inputH,
          (y) => List.generate(inputW, (x) {
            final p = resized.getPixel(x, y);
            return [p.r / 255.0, p.g / 255.0, p.b / 255.0];
          }),
        ),
      );

      // 4) Saída do seu modelo: [1, 25200, 6] = [cx, cy, w, h, conf, classId]
      final output = List.filled(1 * 25200 * 6, 0.0).reshape([1, 25200, 6]);
      _interpreter.run(input, output);

      // 5) Decodificação + filtro por confiança
      const confThresh = 0.60; // suba/abaixe conforme necessário
      final raw = output[0] as List;
      final boxes = <Map<String, dynamic>>[];

      for (var i = 0; i < raw.length; i++) {
        final obj = raw[i] as List;
        final conf = (obj[4] as num).toDouble();
        if (conf < confThresh) continue;

        // YOLOv5 em TFLite costuma usar centro+largura+altura em pixels do input
        final cx = (obj[0] as num).toDouble();
        final cy = (obj[1] as num).toDouble();
        final w = (obj[2] as num).toDouble();
        final h = (obj[3] as num).toDouble();
        final classId = (obj[5] as num).toInt();

        // Converte para (x1,y1,x2,y2) no espaço 640x640
        final x1 = (cx - w / 2).clamp(0, inputW.toDouble());
        final y1 = (cy - h / 2).clamp(0, inputH.toDouble());
        final x2 = (cx + w / 2).clamp(0, inputW.toDouble());
        final y2 = (cy + h / 2).clamp(0, inputH.toDouble());

        boxes.add({
          'x1': x1,
          'y1': y1,
          'x2': x2,
          'y2': y2,
          'confidence': conf,
          'classId': classId,
        });
      }

      // 6) NMS (remove caixas muito sobrepostas e mantém as de maior confiança)
      const iouThresh = 0.50;
      boxes.sort(
        (a, b) =>
            (b['confidence'] as double).compareTo(a['confidence'] as double),
      );
      final picked = <Map<String, dynamic>>[];

      while (boxes.isNotEmpty) {
        final current = boxes.removeAt(0);
        picked.add(current);

        boxes.removeWhere((b) {
          // se for outra classe, não suprime (ajuste se quiser suprimir por classe também)
          if (b['classId'] != current['classId']) return false;
          final iou = _iou(current, b);
          return iou >= iouThresh;
        });
      }

      // 7) Reescala as caixas do espaço 640x640 para o tamanho original da foto
      final scaleX = origW / inputW;
      final scaleY = origH / inputH;
      final detections = picked.map((d) {
        return {
          'x1': d['x1'] * scaleX,
          'y1': d['y1'] * scaleY,
          'x2': d['x2'] * scaleX,
          'y2': d['y2'] * scaleY,
          'confidence': d['confidence'],
          'classId': d['classId'],
        };
      }).toList();

      // Logs
      if (detections.isEmpty) {
        print(
          '⚠️ Nenhum objeto após NMS (threshold=$confThresh, IoU=$iouThresh).',
        );
      } else {
        print('📦 ${detections.length} objeto(s) após NMS.');
        for (final d in detections) {
          print(
            "→ Classe ${d['classId']} | Conf ${((d['confidence'] as double) * 100).toStringAsFixed(1)}% "
            "| x1:${(d['x1'] as double).toStringAsFixed(1)} y1:${(d['y1'] as double).toStringAsFixed(1)} "
            "x2:${(d['x2'] as double).toStringAsFixed(1)} y2:${(d['y2'] as double).toStringAsFixed(1)}",
          );
        }
      }

      return detections;
    } catch (e) {
      print('❌ Erro na inferência: $e');
      return [];
    }
  }

  double _iou(Map<String, dynamic> a, Map<String, dynamic> b) {
    final ax1 = a['x1'] as double, ay1 = a['y1'] as double;
    final ax2 = a['x2'] as double, ay2 = a['y2'] as double;
    final bx1 = b['x1'] as double, by1 = b['y1'] as double;
    final bx2 = b['x2'] as double, by2 = b['y2'] as double;

    final interX1 = ax1 > bx1 ? ax1 : bx1;
    final interY1 = ay1 > by1 ? ay1 : by1;
    final interX2 = ax2 < bx2 ? ax2 : bx2;
    final interY2 = ay2 < by2 ? ay2 : by2;

    final interW = (interX2 - interX1).clamp(0.0, double.infinity);
    final interH = (interY2 - interY1).clamp(0.0, double.infinity);
    final interArea = interW * interH;

    final areaA = (ax2 - ax1) * (ay2 - ay1);
    final areaB = (bx2 - bx1) * (by2 - by1);

    final union = areaA + areaB - interArea;
    if (union <= 0) return 0.0;
    return interArea / union;
  }

  Interpreter get interpreter => _interpreter;
}
