import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { createFileRoute } from '@tanstack/react-router';
import { useEffect, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf'
// Import the required CSS files
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

export const Route = createFileRoute('/blog')({
  component: RouteComponent,
})


export function DocumentComponent({ url } : { url: string }) {
  const [numPages, setNumPages] = useState<number>(0);

  const viewerRef = useRef<HTMLDivElement>(null);
  const [computedWidth, setComputedWidth] = useState(728); 

  useEffect(() => {
    const observer = new ResizeObserver(() => {
      if (viewerRef.current) {
        // Trigger a re-render by updating the memo dependency
        setComputedWidth((viewerRef.current.clientWidth || 728) - 64);
      }
    });

    if (viewerRef.current) {
      observer.observe(viewerRef.current);
    }

    return () => {
      observer.disconnect();
    };
  }, [viewerRef]);

  function onDocumentLoadSuccess({ numPages } : { numPages: number }) {
    setNumPages(numPages);
  }

  return (
    <Card ref={viewerRef} className='bg-secondary-background'>
      <CardHeader></CardHeader>
      <CardContent>
        <Document
          className='m-0 w-full'
          file={{ url }}
          onLoadSuccess={onDocumentLoadSuccess}
          key={crypto.randomUUID()}
        >
          {Array.from(new Array(numPages), (el, index) => (
            <Page
              className='w-full'
              key={`page_${index + 1}`} 
              pageNumber={index + 1} 
              width={computedWidth}
            />
          ))}
        </Document>
      </CardContent>
    </Card>
  )
}

function RouteComponent() {
  return (
    <div className='notebook'>
      <DocumentComponent url='/1.0-building-blocks.pdf' key='/1.0-building-blocks.pdf' />
      <DocumentComponent url='/2.0-cnn-layer.pdf' key='/2.0-cnn-layer.pdf' />
      <DocumentComponent url='/3.0-resnet-architecture.pdf' key='/3.0-resnet-architecture.pdf' />
      <DocumentComponent url='/4.0-data-cleaning-and-collection.pdf' key='/4.0-data-cleaning-and-collection.pdf' />
      <DocumentComponent url='/5.0-training.pdf' key='/5.0-training.pdf' />
    </div>
  )
}