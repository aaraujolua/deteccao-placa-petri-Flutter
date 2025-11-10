// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:deteccao_placa_petri/main.dart';
import 'package:deteccao_placa_petri/tflite_helper.dart';

void main() {
  testWidgets('App carrega a tela inicial corretamente', (
    WidgetTester tester,
  ) async {
    final tflite = TFLiteHelper();
    // Aqui não precisamos carregar o modelo de verdade
    await tester.pumpWidget(MyApp(tflite: tflite));

    // Verifica se o título principal aparece
    expect(find.text('Detecção de Placas'), findsOneWidget);
    expect(find.byIcon(Icons.camera_alt), findsOneWidget);
  });
}
