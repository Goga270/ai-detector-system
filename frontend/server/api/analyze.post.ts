import { spawn } from 'child_process';
import { createError, defineEventHandler, readBody } from 'h3';
import path from 'path';

export default defineEventHandler(async (event) => {
  const body = await readBody(event);

  const { text } = body;

  return new Promise((resolve, reject) => {
    // Вычисляем абсолютные пути
    const rootDir = process.cwd(); // frontend/

    const pythonPath = path.join(rootDir, '../ml/venv/bin/python3');

    const scriptPath = path.join(rootDir, '../ml/src/utils/handler.py');

    const modelsPath = path.join(rootDir, '../ml/models/');

    const workingDir = path.dirname(modelsPath);

    const py = spawn(pythonPath, [scriptPath, text], {
      cwd: workingDir
    });

    let output = '';

    let errorOutput = '';

    py.stdout.on('data', (data) => {
      output += data.toString();
    });
    py.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    py.on('close', (code) => {
      if (code !== 0) {
        console.error('Python Error:', errorOutput);

        return reject(createError({ statusCode: 500, statusMessage: 'Internal ML Error' }));
      }
      try {
        resolve(JSON.parse(output));
      } catch (e) {
        reject(createError({ statusCode: 500, statusMessage: 'Invalid Python Output' }));
      }
    });
  });
});
