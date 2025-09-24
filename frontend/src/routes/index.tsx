import { ActionButton } from '@/components/action-button'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { uploadImage } from '@/features/scanner/api/scanner'
import { AnimalPrediction } from '@/features/scanner/enums/animation-prediction'
import { ScannerStage } from '@/features/scanner/enums/scanner-stage'
import { useMutation } from '@tanstack/react-query'
import { createFileRoute } from '@tanstack/react-router'
import { Link } from '@tanstack/react-router'
import { Aperture, ArrowLeft, Cat, Check, Disc, Dog, HelpCircle, PawPrint } from 'lucide-react'
import { createContext, useContext, useEffect, useRef, useState } from 'react'
import { Camera } from "react-camera-pro";

export const Route = createFileRoute('/')({
  component: RouteComponent,
})

function GoBackBtn({ stage } : { stage: ScannerStage }) {
  const { setStage } = useContext(ScannerContext)
  return (
    <div className='justify-end grid'>
      <Button 
        className='cursor-pointer' 
        onClick={() => setStage(stage)}>
          <ArrowLeft /> Go Back
        </Button>
    </div>
  )
}

function TakeActionComponents({ setImage } : { setImage: (img: File) => void }) {
  const { setStage } = useContext(ScannerContext)
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files && files.length > 0) {
      setImage(files[0])
      setStage(ScannerStage.UPLOAD_IMAGE)
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className='gap-8 grid sm:grid-cols-2'>
        <ActionButton 
          title='Scan'
          subtitle='Take a picture of a potential dog, or cat.'
          icon='scan-search'
          onClick={() => { setStage(ScannerStage.SCAN_IMAGE) }}
        />
        <div>
          <ActionButton 
            title='Upload'
            subtitle='Upload an image instead, nothing will be stored.'
            icon='upload'
            onClick={handleClick}
          />
          {/* Hidden file input */}
          <input 
            type="file"
            ref={fileInputRef}
            onChange={handleFileSelect}
            onClick={(event) => { (event.target as HTMLInputElement).value = '' }}
            style={{ display: 'none' }}
            accept=".png,.jpg,.jpeg,.heic"
          />
        </div>
    </div>
  )
}

function ScanImageComponents({ setImage } : { setImage: (img: File) => void }) {
  const { setStage } = useContext(ScannerContext)
  const camera = useRef(null);

  function onImageSnap() {
    const photo = (camera.current as any).takePhoto()
    setImage(convertBase64ToFile(photo) as File)
    setStage(ScannerStage.UPLOAD_IMAGE)
  }

  // https://forums.meteor.com/t/base64-convert-back-to-file/34188
  function convertBase64ToFile(image: string) {
  const byteString = atob(image.split(',')[1]);
  const ab = new ArrayBuffer(byteString.length);
  const ia = new Uint8Array(ab);
  for (let i = 0; i < byteString.length; i += 1) {
    ia[i] = byteString.charCodeAt(i);
  }
  const newBlob = new Blob([ab], {
    type: 'image/jpeg',
  });

  return newBlob;
};

  return (
    <div className='w-full max-w-sm sm:max-w-lg'>
      <GoBackBtn stage={ScannerStage.TAKE_ACTION} />
      <Card className='mt-4'>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-2xl'><Disc />Camera</CardTitle>
        </CardHeader>
        <CardContent className='relative'>
          <div className='relative border-2 rounded-lg aspect-video overflow-hidden'>
            <Camera 
              ref={camera} 
              facingMode='environment'
              errorMessages={{
              noCameraAccessible: undefined,
              permissionDenied: undefined,
              switchCamera: undefined,
              canvas: undefined
            }} />
          </div>
        </CardContent>
        <CardFooter className='justify-center'>
          <Button onClick={onImageSnap} className='cursor-pointer'><Aperture /> Snap</Button>
        </CardFooter>
      </Card>
    </div>
  )
}

function UploadImageComponents({ image, inferImage } : { image: File, inferImage: () => void }) {
  const [imageUrl, _setImageUrl] = useState(URL.createObjectURL(image));

  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  return (
    <div className='w-full max-w-sm sm:max-w-lg'>
      <GoBackBtn stage={ScannerStage.TAKE_ACTION} />
      <Card className='mt-4'>
        <CardHeader>
          <CardTitle className='flex items-center gap-2 text-2xl'><PawPrint /> Upload Image</CardTitle>
          <CardDescription>Ready to test if it's a dog or a cat?</CardDescription>
        </CardHeader>
        <CardContent className='relative'>
          <div className='relative border-2 rounded-lg aspect-video overflow-hidden'>
            <img 
              src={image ? URL.createObjectURL(image) : ''} 
              alt="Uploaded file" 
              className="w-full h-full object-cover"
            />
          </div>
        </CardContent>
        <CardFooter className='justify-center'>
          <Button className='cursor-pointer' onClick={inferImage}><Check /> Test it</Button>
        </CardFooter>
      </Card>
    </div>
  )
}

function InferImageComponents({ image, result }: { image: File, result: number[] }) {
  const [imageUrl, _setImageUrl] = useState(URL.createObjectURL(image));

  function softmax(inputVector: number[]): number[] {
    const maxVal = Math.max(...inputVector);
    const exponents = inputVector.map(value => Math.exp(value - maxVal));
    const sumExponents = exponents.reduce((sum, current) => sum + current, 0);
    const probabilities = exponents.map(expValue => expValue / sumExponents);

    return probabilities;
  }

  const probabilities = softmax(result);
  const confidenceThreshold = 0.7;
  const confidence = Math.max(...probabilities);
  const confidencePercentage = Math.round(confidence * 100);

  let titleText: string;
  let icon: React.ReactNode;
  let description: string;
  let animalName: string;

  if (probabilities[0] > probabilities[1] && probabilities[0] > confidenceThreshold) {
    prediction = AnimalPrediction.Cat;
    animalName = "cat";
    icon = <Cat className="w-8 h-8" />;
    
    if (confidence >= 0.9) {
      titleText = `This is surely a ${animalName}`;
      description = "The model is super sure that this is a cat. Congratulation! It finds very strong feline traits across the picture. Whiskers, paws, ear shape, and overall body outline match what is usually seen in cats. The confidence is so high that almost every detail agrees with cat features. This level of certainty means the image fits perfectly with what the model has learned to recognize as a cat.";
    } else if (confidence >= 0.8) {
      titleText = `This is likely a ${animalName}`;
      description = "The model is quite sure that this is a cat. Most signs in the picture align with what cats look like. It sees facial shape, body structure, and typical small details that belong to a cat. The confidence is strong but not absolute. There might be small factors such as picture clarity, angle, or background that add a little uncertainty. Still, the signs lean heavily toward this being a cat.";
    } else if (confidence >= 0.7) {
      titleText = `This is maybe a ${animalName}?`;
      description = "The image seems to show feline features but the model has only moderate confidence. It notices some traits such as ear shape, possible whiskers, and outline that suggest a cat, yet these are not very clear. The result is less solid because picture quality, strange pose, or partial view may hide stronger details. The model is not ready to confirm fully but does not ignore the possibility of a cat being present.";
    } else {
      titleText = `Uhhh, could be a ${animalName}`;
      description = "The model detects some signs that point toward a cat but with clear hesitation. The features that appear are faint and may not be enough to confirm. Lighting, blur, or unusual position may make it harder to see the normal cat markers. The model leaves open the chance that this is a cat, but also accepts that the match is weak and uncertain.";
    }
  } else if (probabilities[1] > probabilities[0] && probabilities[1] > confidenceThreshold) {
    prediction = AnimalPrediction.Dog;
    animalName = "dog";
    icon = <Dog className="w-8 h-8" />;
    
    if (confidence >= 0.9) {
      titleText = `This is surely a ${animalName}`;
      description = "The model is very confident that this is a dog. It finds many strong dog features such as snout length, ear position, eye shape, and body form that are clear and consistent. The level of confidence means that most parts of the picture agree with typical canine traits. With this much certainty, the model sees no major contradiction and considers this a clear case of a dog being in the image.";
    } else if (confidence >= 0.8) {
      titleText = `This is likely a ${animalName}`;
      description = "The model finds strong evidence of a dog in the picture. Features such as the face, body outline, and posture are highly similar to common dog shapes. The certainty is high but not absolute, meaning there might be small influences like image angle or light that reduce complete clarity. Still, most of the signs point directly to a dog and the overall confidence is strong.";
    } else if (confidence >= 0.7) {
      titleText = `This is maybe a ${animalName}?`;
      description = "The model notices several dog-like features but with only moderate strength. Some aspects like ear shape or outline may suggest a dog but other details are not as clear. Image conditions, distance, or obstructions might weaken the overall clarity. The chance of this being a dog is real, but the evidence does not fully convince the model.";
    } else {
      titleText = `Uhhh, could be a ${animalName}`;
      description = "The model detects a few possible dog characteristics but is not confident. It sees weak or partial matches such as outline or posture that may belong to a dog but are not strong enough to confirm. External factors like picture quality or unusual context can explain why the result is uncertain. The model accepts the possibility but with heavy doubt.";
    }
  } else {
    prediction = AnimalPrediction.Neither;
    titleText = "This doesn't look like a cat or dog";
    icon = <HelpCircle className="w-8 h-8" />;
    description = "The model cannot say this is a cat or a dog with confidence. The features do not match strongly with either one. This could mean the image shows a different animal such as a rabbit, bird, or something else entirely. It may also be that the picture is unclear, has many objects, or is captured in a way that hides the main subject. Because of this, the model cannot classify the image into either cat or dog with trust.";
  }


  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  return (
    <div className='w-full max-w-sm sm:max-w-lg'>
      <GoBackBtn stage={ScannerStage.TAKE_ACTION} />
      <Card className='mt-4'>
        <CardHeader>
          <CardTitle className='flex items-center gap-3 font-black text-2xl'>
            {icon}
            {titleText}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className='relative border-2 rounded-lg aspect-video overflow-hidden'>
            <img 
              src={image ? URL.createObjectURL(image) : ''} 
              alt="Uploaded file" 
              className="w-full h-full object-cover"
            />
          </div>
          <div className="mt-4 mb-6">
            <div className="flex justify-between items-center mb-2">
              <span className="font-black text-lg">Confidence Level</span>
              <span className="bg-black px-2 py-1 font-black text-white text-lg secondary-background">{confidencePercentage}%</span>
            </div>
            <div className="relative">
              <Progress
                value={confidencePercentage} 
                className="shadow-[2px_2px_0px_0px_#000] border-4 border-black h-6"
              />
            </div>
          </div>

          <div className="bg-pink-200 shadow-[4px_4px_0px_0px_#000] p-4 border-4 border-black">
            <h3 className="mb-2 font-black text-lg">Analysis Details</h3>
            <p className="font-medium leading-relaxed">
              {description}
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}



const ScannerContext = createContext({
  stage: ScannerStage.TAKE_ACTION,
  setStage: (_stage: ScannerStage) => {}
});

function RouteComponent() {
  const [stage, setStage] = useState<ScannerStage>(ScannerStage.TAKE_ACTION)
  const [image, setImage] = useState<File>();

  const mutation = useMutation({
    mutationFn: async () => await uploadImage(image as File),
    onSettled: () => {
      setStage(ScannerStage.INFERENCE)
    }
  })

  function inferImage() {
    mutation.mutate();
    setStage(ScannerStage.LOADING)
  }
  
  return (
    <div className='pb-32'>
      <h1 className='text-3xl lg:text-5xl text-center'>
        Dog and Cat Classifier
      </h1>
      <p className='mt-2 text-xs sm:text-base text-center'>
        Built from scratch using the ResNet-50 architecture. <Link
          className='underline'
          to='/blog'
        >Want to see how?</Link>
      </p>
      <p className='mt-2 text-xs sm:text-base text-center'>*There will be false positives on an image that is neither a dog or a cat. <br />Evan did not train it with a 'neither' label</p>
      <div className='flex justify-center gap-4 mt-4'>
        <ScannerContext.Provider value={{ stage, setStage }}>
          {stage === ScannerStage.TAKE_ACTION && 
            <TakeActionComponents setImage={setImage} />}
          {stage === ScannerStage.SCAN_IMAGE && 
            <ScanImageComponents setImage={setImage} />}
          {stage === ScannerStage.UPLOAD_IMAGE && image && 
            <UploadImageComponents inferImage={inferImage} image={image} />}
          {stage === ScannerStage.LOADING && 
            <div className='mt-16 font-heading text-4xl'>
              Loading...
            </div>
          }
          {stage === ScannerStage.INFERENCE && image && 
            mutation.data && !mutation.isError &&
            <InferImageComponents image={image} result={mutation.data.result} />
          }
        </ScannerContext.Provider>
      </div>
    </div>
  )
}
