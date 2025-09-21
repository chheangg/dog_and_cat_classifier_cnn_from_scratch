import { ActionButton } from '@/components/action-button'
import { createFileRoute } from '@tanstack/react-router'
import { Link } from '@tanstack/react-router'

export const Route = createFileRoute('/')({
  component: RouteComponent,
})

function RouteComponent() {
  return (
    <div>
      <h1 className='text-3xl lg:text-5xl text-center'>
        Dog and Cat Classifier
      </h1>
      <p className='mt-2 text-center'>
        Built from scratch using the ResNet-50 architecture. <Link
          className='underline'
          to='/blog'
        >Want to see how?</Link>
      </p>
      <div className='flex justify-center gap-4 md:gap-8 mt-8 md:mt-16'>
        <ActionButton 
          title='Scan'
          subtitle='Take a picture of a potential dog, or cat.'
          icon='scan-search'
        />
        <ActionButton 
          title='Upload'
          subtitle='Upload an image instead, nothing will be stored.'
          icon='upload'
        />
      </div>
    </div>
  )
}
