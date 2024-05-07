using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using System.IO;
using Newtonsoft.Json;
using System.Text;
using TMPro;
using UnityEngine.UI;

/*
 *              Whisper Inference Code
 *              ======================
 *  
 *  Put this script on the Main Camera
 *  
 *  In Assets/StreamingAssets put:
 *  
 *  AudioDecoder_Tiny.sentis
 *  AudioEncoder_Tiny.sentis
 *  LogMelSepctro.sentis
 *  vocab.json
 * 
 *  Drag a 30s 16khz mono uncompressed audioclip into the audioClip field. 
 * 
 *  Install package com.unity.nuget.newtonsoft-json from packagemanger
 *  Install package com.unity.sentis
 * 
 */


public class RunWhisper : MonoBehaviour
{
    private IWorker decoderEngine, encoderEngine, spectroEngine;

    private const BackendType backend = BackendType.GPUCompute;

    // Link your audioclip here. Format must be 16Hz mono non-compressed.
    public AudioClip audioClip;

    // This is how many tokens you want. It can be adjusted.
    private const int maxTokens = 100;

    //Special tokens see added tokens file for details
    private const int END_OF_TEXT = 50257;
    private const int START_OF_TRANSCRIPT = 50258;
    private const int ENGLISH = 50259;
    private const int GERMAN = 50261;
    private const int FRENCH = 50265;
    private const int TRANSCRIBE = 50359; //for speech-to-text in specified language
    private const int TRANSLATE = 50358;  //for speech-to-text then translate to English
    private const int NO_TIME_STAMPS = 50363;
    private const int START_TIME = 50364;

    private int numSamples;
    private float[] data;
    private string[] tokens;

    private int currentToken = 0;
    private int[] outputTokens = new int[maxTokens];

    // Used for special character decoding
    private int[] whiteSpaceCharacters = new int[256];

    private TensorFloat encodedAudio;

    private bool transcribe = false;
    private string outputString = "";

    // Maximum size of audioClip (30s at 16kHz)
    private const int maxSamples = 30 * 16000;

    /// <summary>
    /// サンプル音声を再生するボタン
    /// </summary>
    [SerializeField] private Button playSampleVoiceButton;

    /// <summary>
    /// 文字起こし開始ボタン
    /// </summary>
    [SerializeField] private Button playTranscribeVoiceButton;

    /// <summary>
    /// 翻訳された音声のテキスト
    /// </summary>
    [SerializeField] private TextMeshProUGUI transcribeText;

    /// <summary>
    /// サンプル音声
    /// </summary>
    [SerializeField] private AudioSource _audioSource;

    private void Start()
    {
        SetupWhiteSpaceShifts();

        GetTokens();

        Model decoder = ModelLoader.Load(Application.streamingAssetsPath + "/AudioDecoder_Tiny.sentis");

        Model decoderWithArgMax = Functional.Compile(
            (tokens, audio) => Functional.ArgMax(decoder.Forward(tokens, audio)[0], 2),
            (decoder.inputs[0], decoder.inputs[1])
        );

        Model encoder = ModelLoader.Load(Application.streamingAssetsPath + "/AudioEncoder_Tiny.sentis");
        Model spectro = ModelLoader.Load(Application.streamingAssetsPath + "/LogMelSepctro.sentis");

        decoderEngine = WorkerFactory.CreateWorker(backend, decoderWithArgMax);
        encoderEngine = WorkerFactory.CreateWorker(backend, encoder);
        spectroEngine = WorkerFactory.CreateWorker(backend, spectro);

        outputTokens[0] = START_OF_TRANSCRIPT;
        outputTokens[1] = ENGLISH;// GERMAN;//FRENCH;//
        outputTokens[2] = TRANSCRIBE; //TRANSLATE;//
        outputTokens[3] = NO_TIME_STAMPS;// START_TIME;//
        currentToken = 3;


        playSampleVoiceButton.onClick.AddListener(() =>
        {
            _audioSource.clip = audioClip;
            _audioSource.Play();
        });

        playTranscribeVoiceButton.onClick.AddListener(() =>
        {
            LoadAudio();
            EncodeAudio();
            transcribe = true;
        });
    }

    private void LoadAudio()
    {
        if(audioClip.frequency != 16000)
        {
            Debug.Log($"The audio clip should have frequency 16kHz. It has frequency {audioClip.frequency / 1000f}kHz");
            return;
        }

        numSamples = audioClip.samples;

        if (numSamples > maxSamples)
        {
            Debug.Log($"The AudioClip is too long. It must be less than 30 seconds. This clip is {numSamples/ audioClip.frequency} seconds.");
            return;
        }

        data = new float[maxSamples];
        numSamples = maxSamples;
        //We will get a warning here if data.length is larger than audio length but that is OK
        audioClip.GetData(data, 0);
    }


    private void GetTokens()
    {
        var jsonText = File.ReadAllText(Application.streamingAssetsPath + "/vocab.json");
        var vocab = Newtonsoft.Json.JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonText);
        tokens = new string[vocab.Count];
        foreach(var item in vocab)
        {
            tokens[item.Value] = item.Key;
        }
    }

    private void EncodeAudio()
    {
        using var input = new TensorFloat(new TensorShape(1, numSamples), data);

        spectroEngine.Execute(input);
        var spectroOutput = spectroEngine.PeekOutput() as TensorFloat;

        encoderEngine.Execute(spectroOutput);
        encodedAudio = encoderEngine.PeekOutput() as TensorFloat;
    }


    // Update is called once per frame
    private void Update()
    {
        if (transcribe && currentToken < outputTokens.Length - 1)
        {
            using var tokensSoFar = new TensorInt(new TensorShape(1, outputTokens.Length), outputTokens);

            var inputs = new Dictionary<string, Tensor>
            {
                {"input_0", tokensSoFar },
                {"input_1", encodedAudio }
            };

            decoderEngine.Execute(inputs);
            var tokensPredictions = decoderEngine.PeekOutput() as TensorInt;

            tokensPredictions.CompleteOperationsAndDownload();

            int ID = tokensPredictions[currentToken];

            outputTokens[++currentToken] = ID;

            if (ID == END_OF_TEXT)
            {
                transcribe = false;
            }
            else if (ID >= tokens.Length)
            {
                outputString += $"(time={(ID - START_TIME) * 0.02f})";
            }
            else outputString += GetUnicodeText(tokens[ID]);

            Debug.Log(outputString);
            transcribeText.text = outputString;
        }
    }

    // Translates encoded special characters to Unicode
    private string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    private string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += ((int)letter <= 256) ? letter :
                (char)whiteSpaceCharacters[(int)(letter - 256)];
        }
        return outText;
    }

    private void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
        {
            if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
        }
    }

    private bool IsWhiteSpace(char c)
    {
        return !(('!' <= c && c <= '~') || ('�' <= c && c <= '�') || ('�' <= c && c <= '�'));
    }

    private void OnApplicationQuit()
    {
        if (Input.GetKeyDown(KeyCode.Escape)) Application.Quit();
    }

    private void OnDestroy()
    {
        decoderEngine?.Dispose();
        encoderEngine?.Dispose();
        spectroEngine?.Dispose();
    }
}
